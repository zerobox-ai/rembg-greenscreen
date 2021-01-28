param(
    [string]$target,    
[int]$batches=10,
[int]$concurrency=4,
[int]$fps=30,
[bool]$skip_create_frames=0,
[bool]$fix_in_place=0)

if( !$fix_in_place){
    
    $tempfolder = $target -replace ".mp4", ""

    mkdir ./$tempfolder -Force
    cp $target ./$tempfolder
    cd $tempfolder

    $folder = Get-Location

    # turn video into frames
    if(-not $skip_create_frames){
        ffmpeg -i $target -vf fps=$fps %d.jpg
    }

}

# retry failed ones
1..3 | % { 

    # delete any failed png conversions
    ls *.png | where Length -eq 0 | del

    $d = 0;

    # process all the frames into batches
    Get-ChildItem | 
        Where-Object Name -imatch "jpg" |
        ForEach-Object { 
            New-Object PSObject  -property @{
                Test= -not (Test-Path -Path ($_.name -replace "jpg", ".out.png") ); 
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

#check for every JPG file there is a corresponding non-empty PNG present
if( (Get-ChildItem | 
    Where-Object Name -imatch "jpg" |
    ForEach-Object { 
        New-Object PSObject  -property @{
            PngNotCreated = -not (Test-Path -Path ($_.name -replace "jpg", ".out.png") ); 
            PngEmpty = -not (Test-Path -Path ($_.name -replace "jpg", ".out.png") ) -or (get-item ($_.name -replace "jpg", ".out.png")).Length -eq 0;
        } } | 
        Where-Object {$_.PngNotCreated -or $_.PngEmpty }|
        Measure-Object).Length -eq 0 ) {

            del *.jpg
    
            ffmpeg -i %d.out.png -vcodec png ($target -replace ".mp4", ".mov")
        
            Move-Item *.mov ../

            del *.png

            cd ../
            del $tempfolder -Recurse
}

else{
    cd ../
}

# go back

