param(
[string]$target,    
[int]$retry_times=5)

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
cp $target $folder/$tempfolder
Set-Location $folder/$tempfolder

# have we already exported the frames?
$skip_create_frames = (Get-ChildItem *.jpg | Measure-Object).Count -gt 10

# turn video into frames
if(-not $skip_create_frames){
    Remove-Item *.jpg
    # now halfing the res of the alpha matte, gives a pretty big speedup
    ffmpeg -i $target -vf "fps=$fps,scale=iw*.5:ih*.5" %d.jpg
}else{
    Write-Output "skipping creation of frames..."
}

# retry failed ones
1..$retry_times | % { 

    Write-Output "deleting failed conversions"

    # delete any failed png conversions
    Get-ChildItem *.png | where Length -eq 0 | Remove-Item

    Write-Output "converting JPG->PNG with virtual green screen"

    if($PSVersionTable.Platform -ne "Unix"){
        C:/Windows/System32/cmd.exe /K "$env:anaconda\\Scripts\\activate.bat $env:anaconda && cd $env:rembg && python -m src.rembg.cmd.cli -p $folder\$tempfolder\ 2>&1 && exit"
    }else{
        rembg -p $folder/$tempfolder/
    }

    $no_pngs = (Get-ChildItem *.png | Measure-Object).Count

    # num jpgs = num pngs
    $equal_pngs = $no_pngs -gt 0 -and $no_pngs -eq (Get-ChildItem *.jpg | Measure-Object).Count
    # none of the conversations failed
    $noempty_pngs = $no_pngs -gt 10 -and (Get-ChildItem *.png | Where-Object Length -eq 0 | Measure-Object ).Count -eq 0

    #check for every JPG file there is a corresponding non-empty PNG present and there are some pngs ready to go
    if( $equal_pngs -and $noempty_pngs){

        echo "equal jpgs and pngs detected, and all pngs are non-zero, making mov file..."

        ffmpeg -i %d.out.png -vcodec png ($target -replace "\..+", ".mov")

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
}

Write-Output "didnt detect conditions to create mov (tried several times)"
# go back
Set-Location $orig_folder