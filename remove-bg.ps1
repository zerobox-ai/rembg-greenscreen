param(
[string]$target,    
[int]$retry_times=5, 
[int]$highquality=1)

# note that this has been superceded by the streaming version in this repo
# I am leaving this here for people who want to do it the manual ->JPG->FFMPEG 
# way using Daniel's original version

# does the target even exist?
if( -not(Test-Path $target) ){
    Write-Output ("Can't find file {0}" -f $target)
    exit
}

# if there are PNGs in this folder you probably called this script wrong
if( test-path "./*.png" ){
    Write-Output ("PNGs detected here, you probably messed up")
    exit
}

# do we already have the .MOV file?
if( test-path "{0}.mov" -f (Get-ChildItem $target).BaseName ){
    Write-Output ("MOV file already exists for {0}" -f $target)
    exit
}

# compute the fps dynamically, so we don't waste compute on setting higher
$fps_fraction = ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate $target
$fps = [math]::ceiling( (Invoke-Expression $fps_fraction) )

# if the file is VBR you get crazy results like 16000
if( $fps -gt 30 -or $fps -lt 0 ){
    $fps=24
}

$orig_folder = Get-Location

$tempfolder = "f_{0}" -f (get-item $target).Name

Write-Output $tempfolder

$folder = (get-item $target).Directory.FullName

New-Item -Path "$folder" -Name "$tempfolder" -ItemType "directory" -Force
Copy-Item $target $folder/$tempfolder
Set-Location $folder/$tempfolder

# have we already exported the frames?
$count_jpg_filename = "count_jpgs.txt"
$skip_create_frames = Test-Path $count_jpg_filename

# turn video into frames
if(-not $skip_create_frames){
    Remove-Item *.jpg
    # making the resolutuon of the images as small as possible, the NN only takes in 320^2
    # this gives us a significant speed up on IO and preprocessing
    if($highquality){
        ffmpeg -i $target -vf "fps=30,scale=-1:320" %d.bmp
    }
    else{
        ffmpeg -i $target -vf "fps=30,scale=-1:320" -qscale:v 2 %d.jpg
    }
    
    #we need to record how many jpgs there are
    $count_jpgs = (Get-ChildItem *.jpg | Measure-Object).Count
    $count_jpgs | out-file -FilePath $count_jpg_filename

}else{
    $count_jpgs = [int](Get-Content $count_jpg_filename)
    Write-Output "skipping creation of frames..."
}

if($count_jpgs -lt 100){
    Write-Output ("Exiting... looks like there are not enough JPGs for this video in {0}" -f $count_jpg_filename)
    exit
}

# retry failed ones
1..$retry_times | % { 

    Write-Output "deleting failed conversions"

    # delete any failed png conversions
    Get-ChildItem *.png | where Length -eq 0 | Remove-Item

    # are there any .out.out pngs? This means the rembg needs the glob update applied
    if( Test-Path *.out.out*.png ){
        Write-Output "*.out.out*.png detected, fix your rembg -- deleting and exiting"
        Remove-Item (Get-ChildItem *.out.out*.png)
        exit
    }

    # we want this to run incrementally i.e. what if this was stopped previously
    # half way through the conversion, solution is to delete all jpgs which already
    # have corresponding non-zero pngs 

    $pngs = @{}
    $todelete = New-Object Collections.Generic.List[string]

    Get-ChildItem *.png | ForEach-Object{ $pngs[$_.BaseName]=1 }
    Get-ChildItem *.jpg | ForEach-Object{ 
        if( $pngs["{0}.out" -f $_.BaseName] -eq 1 ) {
            $todelete.Add($_.Name)
        } 
    }

    # assuming this will be faster as an atomic operation
    if( $todelete.Count -gt 0 ) {
        Remove-Item $todelete
        Write-Output ("Removed {0} JPGs as the PNGs were already good to go" -f $todelete.Count)
    }

    Write-Output "converting JPG->PNG with virtual green screen"

    if($PSVersionTable.Platform -ne "Unix"){
        C:/Windows/System32/cmd.exe /K "$env:anaconda\\Scripts\\activate.bat $env:anaconda && cd $env:rembg && python -m src.rembg.cmd.cli -p $folder\$tempfolder\ 2>&1 && exit"
    }else{
        rembg -p $folder/$tempfolder/
    }

    $no_pngs = (Get-ChildItem *.png | Measure-Object).Count

    # num jpgs = num pngs, #we use the cached version of $count_jpgs as we are deleting them on the fly
    $equal_pngs = $no_pngs -gt 0 -and $no_pngs -eq $count_jpgs
    # none of the conversations failed
    $noempty_pngs = $no_pngs -gt 10 -and (Get-ChildItem *.png | Where-Object Length -eq 0 | Measure-Object ).Count -eq 0

    #check for every JPG file there is a corresponding non-empty PNG present and there are some pngs ready to go
    if( $equal_pngs -and $noempty_pngs){

        Write-Output "equal jpgs and pngs detected, and all pngs are non-zero, making mov file..."

        ffmpeg -i %d.out.png -vcodec png ("{0}.mov" -f (Get-ChildItem $target).BaseName)

        if( Test-Path *.mov ) {

            Write-Output "moving mov file back to main dir"
            Move-Item *.mov ../
            Set-Location ..
            Remove-Item $tempfolder -Recurse -Force
            Set-Location $orig_folder

        } else{
            Write-Output "no MOV found"
        }

         # we good
         exit
    }
    else{
        Write-Output "didn't detect equal jpg-png"
    }
}

Write-Output "didnt detect conditions to create mov (tried several times)"
# go back
Set-Location $orig_folder