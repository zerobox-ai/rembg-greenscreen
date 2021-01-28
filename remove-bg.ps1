param(
    [string]$target,    
[int]$batches=10,
[int]$concurrency=4,
[int]$fps=30,
[int]$retry_times=2)


$tempfolder = $target -replace ".mp4", ""

mkdir ./$tempfolder -Force
cp $target ./$tempfolder
cd $tempfolder

$folder = Get-Location

# have we already exported the frames?
$skip_create_frames = (ls *.png | measure).Count -gt 10

# turn video into frames
if(-not $skip_create_frames){
    ffmpeg -i $target -vf fps=$fps %d.jpg
}else{
    echo "skipping creation of frames..."
}

# retry failed ones
1..$retry_times | % { 

    # delete any failed png conversions
    ls *.png | where Length -eq 0 | del

    $d = 0;

    # process all the frames into batches
    Get-ChildItem | 
        Where-Object Name -imatch "jpg" |
        ForEach-Object { 
            New-Object PSObject  -property @{
                Test= (Test-Path -Path ($_.name -replace "jpg", ".out.png") ); 
                Path=$_.Name;
                Number=$d++
            } } | 
            Where-Object Test | 
            Group-Object -Property {$_.Number % $batches} |
            ForEach-Object{
                $gn = $_.Name
                mkdir $gn -Force

                $_.Group | ForEach-Object{ Copy-Item $_.Path $gn }

                Start-ThreadJob -ThrottleLimit $concurrency -ScriptBlock {param([string]$folder, [string]$gn)

                    C:/Windows/System32/cmd.exe /K "$env:anaconda\\Scripts\\activate.bat $env:anaconda && cd $env:rembg && python -m src.rembg.cmd.cli -p $folder\$gn 2>&1 && exit"
                
                    Copy-Item $folder/$gn/*.png $folder
                    Remove-Item $folder/$gn/ -Force -Recurse

                } -ArgumentList "$folder", "$gn"

            }

    get-job | Wait-Job

}

# num jpgs = num pngs
$equal_pngs = (ls *.jpg | measure).Count -eq (ls *.png | measure).Count
# none of the conversations failed
$noempty_pngs = (ls *.png | where Length -eq 0 | measure ).Count -eq 0


#check for every JPG file there is a corresponding non-empty PNG present
if( $equal_pngs -and $noempty_pngs){

    echo "equal jpgs and pngs detected, and all pngs are non-zero, making mov file..."

    del *.jpg

    ffmpeg -i %d.out.png -vcodec png ($target -replace ".mp4", ".mov")

    echo "moving mov file back to main dir"
    Move-Item *.mov ../

    del *.png
    cd ../
    del $tempfolder -Recurse
}

else{
    cd ../
}

# go back

